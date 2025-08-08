import signal
import sys
import time
import threading
import docker

from loguru import logger as log
from typing import List, Dict, Optional, Union, Any

from docker.errors import DockerException, ContainerError, ImageNotFound, APIError
from docker.models.containers import Container
from docker.models.networks import Network


PGVECTOR_IMAGE = 'pgvector/pgvector:pg17'
CHROMA_DB_IMAGE = 'chromadb/chroma:latest'

SYL_BASE_IMAGE_NAME = 'ohtz/syl:base-latest'
SYL_SERVER_IMAGE_NAME = 'ohtz/syl:server-latest'
SYL_INDEX_IMAGE_NAME = 'ohtz/syl:index-latest'
SYL_WATCHER_IMAGE_NAME = 'ohtz/syl:watcher-latest'

# All syl images for checking updates
SYL_IMAGES = [
    SYL_BASE_IMAGE_NAME,
    SYL_SERVER_IMAGE_NAME,
    SYL_INDEX_IMAGE_NAME,
    SYL_WATCHER_IMAGE_NAME,
]

SYL_SERVER_NAME = 'syl-server'
SYL_NETWORK_NAME = 'syl-network'

SYL_INDEX_PREFIX = 'syl-index-'
SYL_PGVECTOR_PREFIX = 'syl-pgvector-'
SYL_CHROMADB_PREFIX = 'syl-chromadb-'
SYL_WATCHER_PREFIX = 'syl-watcher-'

DEFAULT_CIDR = '172.20.0.0/16'


class DockerManager:
    def __init__(self, network_name: str = SYL_NETWORK_NAME, server_container_name: str = SYL_SERVER_NAME):
        try:
            self.client = docker.from_env()
            self.client.ping()
        except DockerException as e:
            log.error(f'Failed to connect to Docker: {e}')
            raise

        self.shutdown_event = threading.Event()
        self.network_name = network_name
        self.server_container_name = server_container_name

    def container_is_running(self, name: str) -> bool:
        container = self.get_container_by_name(name)
        if container:
            return container.status == 'running'
        return False

    @staticmethod
    def container_is_auto_remove(container: Container) -> bool:
        return container.attrs.get('HostConfig', {}).get('AutoRemove', False)

    def list_containers(self) -> List[Container]:
        return self.client.containers.list(filters={'network': self.network_name})

    def list_indexing_container(self) -> List[Container]:
        return self.client.containers.list(filters={'network': self.network_name, 'name': f'{SYL_INDEX_PREFIX}*'})

    def list_containers_by_label(self, label: str) -> List[Container]:
        return self.client.containers.list(filters={'network': self.network_name, 'label': label})

    def get_container_by_name(self, name: str) -> Optional[Container]:
        containers = self.client.containers.list(
            filters={
                'network': self.network_name,
                'name': f'^{name}$',  # Partial-matches by default, which wasted 15 minutes of my life.. use re
            }
        )

        if len(containers) == 0:
            return None
        else:
            return containers[0]

    def get_containers_by_name_suffix(self, suffix: str) -> Optional[List[Container]]:
        containers = self.client.containers.list(filters={'network': self.network_name, 'name': f'.+-{suffix}$'})
        if len(containers) == 0:
            return None
        return containers

    def get_datastore_containers_by_name_suffix(self, suffix: str) -> Optional[List[Container]]:
        containers = self.client.containers.list(
            filters={
                'network': self.network_name,
                #'name': f'^(?!{SYL_INDEX_PREFIX})(?!{SYL_WATCHER_PREFIX}).+-{suffix}$'  # Doesn't seem like Docker supports negative-lookaheads..
                'name': f'.+-{suffix}$',
            }
        )
        if len(containers) == 0:
            return None
        return [
            c
            for c in containers
            if not c.name.startswith(SYL_INDEX_PREFIX) and not c.name.startswith(SYL_WATCHER_PREFIX)
        ]

    def get_container_ip(self, name: str) -> Optional[str]:
        try:
            container = self.client.containers.get(name)
            return self.get_container_ip_from_container(container)

        except docker.errors.NotFound:
            raise docker.errors.NotFound(f"Container '{name}' not found")
        except docker.errors.APIError as e:
            raise docker.errors.APIError(f'Docker API error: {e}')

    @staticmethod
    def get_container_ip_from_container(container: Container) -> Optional[str]:
        syl_ip = container.attrs.get('NetworkSettings', {}).get('Networks', {}).get('syl-network', {}).get('IPAddress')
        return syl_ip

    def create_container(
        self,
        image: str,
        name: Optional[str] = None,
        daemon_mode: bool = False,
        ports: Optional[Dict[Union[str, int], Union[str, int]]] = None,
        volumes: Optional[Dict[str, str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        extra_args: Optional[List[str]] = None,
        script_args: Optional[List[str]] = None,
        auto_remove: bool = True,
    ) -> Optional[Container]:
        try:
            port_bindings = {}
            if ports:
                for host_port, container_port in ports.items():
                    port_bindings[str(container_port)] = str(host_port)  # To str

            volume_bindings = {}
            if volumes:
                for host_path, container_path in volumes.items():
                    volume_bindings[host_path] = {'bind': container_path, 'mode': 'rw'}

            # script_args are passed as arguments to the entrypoint
            command = script_args

            run_kwargs: Dict[str, Any] = {
                'image': image,
                'name': name,
                'detach': daemon_mode,
                'ports': port_bindings,
                'volumes': volume_bindings,
                'environment': env_vars,
                'command': command,
                'network': self.network_name,
                'auto_remove': auto_remove,
            }

            log.info('Starting container', extra={'run_kwargs': run_kwargs})

            if extra_args:
                for arg in extra_args:
                    if '=' in arg:
                        key, value = arg.split('=', 1)

                        if value.lower() in ('true', 'false'):
                            run_kwargs[key] = value.lower() == 'true'
                        else:
                            run_kwargs[key] = value

            if daemon_mode:
                container = self.client.containers.run(**run_kwargs)
                return self._verify_daemon_container(container, name)
            else:
                run_kwargs['detach'] = True  # Always detach so we can manage output
                container = self.client.containers.create(**run_kwargs)

                container.start()
                return self._handle_non_daemon_container(container, name)

        except ImageNotFound:
            log.error(f'Image not found: {image}')
            return None
        except ContainerError as e:
            log.error(f'Container failed to start: {e}')
            return None
        except APIError as e:
            log.error(f'Docker API error: {e}')
            return None
        except Exception as e:
            log.error(f'Unexpected error starting container: {e}')
            return None

    def _handle_non_daemon_container(self, container: Container, name: Optional[str] = None) -> Optional[Container]:
        def signal_handler(signum, frame):
            log.warning(
                'Received interrupt signal. Stopping container.',
                extra={
                    'name': name,
                    'container_id': container.id,
                },
            )
            try:
                container.stop(timeout=10)
                log.info(f'Container {name or container.short_id} stopped gracefully')
            except Exception as e:
                log.warning(f'Error stopping container: {e}')
                try:
                    container.kill()
                    log.info(f'Container {name or container.short_id} killed forcefully')
                except Exception as e2:
                    log.error(f'Error killing container: {e2}')
            sys.exit(130)  # SIGINT

        original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            for line in container.logs(stream=True, follow=True, stdout=True, stderr=True):
                output = line.decode('utf-8').rstrip()
                if output:
                    print(output, flush=True)

            result = container.wait()

            exit_code = result['StatusCode']

            if exit_code == 0:
                log.info(f'Container {name or container.short_id} completed successfully')
                return container
            else:
                log.error(f'Container {name or container.short_id} failed with exit code: {exit_code}')
                self._cleanup_container(container)
                return None

        except KeyboardInterrupt:
            # This shouldn't happen since we have a signal handler, but just in case
            signal_handler(signal.SIGINT, None)
        except Exception as e:
            log.error(f'error handling container: {e}')
            self._cleanup_container(container)
            return None
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            return None

    def _verify_daemon_container(self, container: Container, name: Optional[str] = None) -> Optional[Container]:
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                container.reload()

                if container.status == 'running':
                    log.info(f'Container {name or container.short_id} is running successfully')
                    return container

                elif container.status in ['exited', 'dead']:
                    log.error(f'Container {name or container.short_id} failed to start (status: {container.status})')
                    break

            except Exception as e:
                log.warning(f'Error checking container status: {e}')

            if attempt < max_attempts - 1:
                time.sleep(0.5)

        self._cleanup_container(container)
        return None

    def _cleanup_container(self, container: Container):
        try:
            container.remove(force=True)
            log.info(f'Cleaned up failed container {container.short_id}')
        except Exception as e:
            log.warning(f'Failed to cleanup container {container.short_id}: {e}')

    def stop_container_by_name(self, name: str):
        container = self.get_container_by_name(name)
        if container:
            self.stop_container(container)

    @staticmethod
    def stop_container(container: Container, timeout: int = 10) -> bool:
        try:
            container.stop(timeout=timeout)
            log.info(f'Container {container.name or container.short_id} stopped successfully')
            return True
        except Exception as e:
            log.error(f'Error stopping container: {e}')
            return False

    def stop_all_containers(self):
        log.info('Stopping all containers..')

        for i, container in enumerate(self.list_containers()):
            try:
                # container.reload()
                self.stop_container(container)
            except Exception as e:
                print(f'Error stopping container {self.container_names[i]}: {e}')

    def get_container_logs(
        self, name: str, max_results: int = 10, tail: bool = False, follow: bool = False
    ) -> Optional[str]:
        """Get logs from a container by name"""
        container = self.get_container_by_name(name)
        if not container:
            log.error(f'Container {name} not found')
            return None

        try:
            if tail and follow:
                log.debug(f'Tailing logs for container {name}. Press Ctrl+C to stop.')
                for line in container.logs(stream=True, follow=True, stdout=True, stderr=True, tail=max_results):
                    print(line.decode('utf-8').rstrip())
                return None
            else:
                logs = container.logs(tail=max_results, stdout=True, stderr=True)
                return logs.decode('utf-8')
        except Exception as e:
            log.error(f'Error getting logs for container {name}: {e}')
            return None

    def get_network(self) -> Optional[Network]:
        networks = self.client.networks.list(filters={'name': self.network_name})
        if not networks or len(networks) == 0:
            log.warning('No networks found')
            return None
        elif len(networks) > 1:
            log.warning(
                'error expected at most one network',
                extra={
                    'network_name': self.network_name,
                    'num_networks': len(networks),
                    'network_names': [n.name for n in networks],
                },
            )
            return None
        return networks[0]

    def get_network_cidr(self) -> Optional[str]:
        network = self.get_network()
        if not network:
            return None
        # .[0].IPAM.Config[0].Subnet
        return network.attrs['IPAM']['Config'][0]['Subnet']

    def network_is_up(self) -> bool:
        try:
            self.client.networks.get(self.network_name)
            return True

        except docker.errors.NotFound:
            return False

        except Exception as e:
            log.error('error checking network status', extra={'error': str(e)})
            return False

    def create_network(self, driver: str = 'bridge', subnet: str = DEFAULT_CIDR) -> bool:
        try:
            if self.network_is_up():
                log.info('Network already exists', extra={'network_name': self.network_name})
                return True

            ipam_pool = docker.types.IPAMPool(subnet=subnet)
            ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])

            log.info('Creating network', extra={'network_name': self.network_name, 'driver': driver})

            self.client.networks.create(name=self.network_name, driver=driver, ipam=ipam_config)

            log.info('Network created successfully', extra={'network_name': self.network_name})
            return True

        except APIError as e:
            log.error('Error creating network', extra={'network_name': self.network_name, 'error': str(e)})
            return False

        except Exception as e:
            log.error('Unexpected error creating network', extra={'error': str(e)})
            return False

    def remove_network(self):
        try:
            log.info(f'Removing network: {self.network_name}')

            network = self.get_network()
            if not network:
                log.error('Network does not exist', extra={'network_name': self.network_name})
                return

            network.remove()
            log.info(f'Network "{self.network_name}" removed successfully')

        except docker.errors.NotFound:
            log.info(f'Network "{self.network_name}" already removed')

        except Exception as e:
            log.error('error removing network', extra={'network_name': self.network_name, 'error': str(e)})

    def get_local_image_digest(self, image_name: str) -> Optional[str]:
        """Get the digest of a local image"""
        try:
            image = self.client.images.get(image_name)
            repo_digests = image.attrs.get('RepoDigests', [])
            if repo_digests:
                return repo_digests[0].split('@')[1] if '@' in repo_digests[0] else None
            return None
        except docker.errors.ImageNotFound:
            return None
        except Exception as e:
            log.error(f'Error getting local image digest for {image_name}: {e}')
            return None

    def get_remote_image_digest(self, image_name: str) -> Optional[str]:
        """Get the digest of the latest remote image from Docker Hub"""
        try:
            import requests

            if ':' in image_name:
                repo_name, tag = image_name.rsplit(':', 1)
            else:
                repo_name, tag = image_name, 'latest'

            auth_url = f'https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo_name}:pull'
            auth_response = requests.get(auth_url, timeout=10)
            if auth_response.status_code != 200:
                return None

            token = auth_response.json().get('token')
            if not token:
                return None

            # Get manifest digest
            manifest_url = f'https://registry-1.docker.io/v2/{repo_name}/manifests/{tag}'
            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.docker.distribution.manifest.v2+json',
            }

            manifest_response = requests.get(manifest_url, headers=headers, timeout=10)
            if manifest_response.status_code != 200:
                return None

            return manifest_response.headers.get('Docker-Content-Digest')

        except Exception as e:
            log.debug(f'Error getting remote image digest for {image_name}: {e}')
            return None

    def is_image_outdated(self, image_name: str) -> bool:
        """Check if local image is outdated compared to remote"""
        local_digest = self.get_local_image_digest(image_name)
        if not local_digest:
            log.error('nope')
            return True  # No local image, needs pull

        remote_digest = self.get_remote_image_digest(image_name)
        if not remote_digest:
            log.error('yep')
            return False  # Can't check remote, assume local is fine

        return local_digest != remote_digest

    def pull_image(self, image_name: str) -> bool:
        """Pull the latest version of an image"""
        try:
            log.info(f'Pulling image: {image_name}')
            self.client.images.pull(image_name)
            log.info(f'Successfully pulled image: {image_name}')
            return True
        except Exception as e:
            log.error(f'Error pulling image {image_name}: {e}')
            return False

    def check_and_pull_outdated_images(self) -> Dict[str, Dict[str, bool]]:
        """Check all images and pull only outdated ones"""
        results = {}
        for image_name in SYL_IMAGES:
            is_outdated = self.is_image_outdated(image_name)
            pulled = False

            if is_outdated:
                pulled = self.pull_image(image_name)

            results[image_name] = {'outdated': is_outdated, 'pulled': pulled}

        return results

    def pull_all_syl_images(self) -> Dict[str, bool]:
        """Pull all syl images regardless of status"""
        results = {}
        for image_name in SYL_IMAGES:
            results[image_name] = self.pull_image(image_name)
        return results


def datastore_name_from_container_name(container_name: str) -> str:
    if container_name.startswith(SYL_WATCHER_PREFIX):
        return '-'.join(container_name.removeprefix(SYL_WATCHER_PREFIX).split('-')[:-1])

    return (
        container_name.removeprefix(SYL_INDEX_PREFIX)
        .removeprefix(SYL_PGVECTOR_PREFIX)
        .removeprefix(SYL_CHROMADB_PREFIX)
    )


def is_server_container(container: Container) -> bool:
    if container.name is None:
        return False
    return container.name == SYL_SERVER_NAME


def is_index_container(container: Container) -> bool:
    if container.name is None:
        return False
    return container.name.startswith(SYL_INDEX_PREFIX)


def is_pgvector_container(container: Container) -> bool:
    if container.name is None:
        return False
    return container.name.startswith(SYL_PGVECTOR_PREFIX)


def is_chromadb_container(container: Container) -> bool:
    if container.name is None:
        return False
    return container.name.startswith(SYL_CHROMADB_PREFIX)


def is_datastore_container(container: Container) -> bool:
    if container.name is None:
        return False
    return container.name.startswith(SYL_PGVECTOR_PREFIX) or container.name.startswith(SYL_CHROMADB_PREFIX)


def is_watcher_container(container: Container) -> bool:
    if container.name is None:
        return False
    return container.name.startswith(SYL_WATCHER_PREFIX)
