from loguru import logger as log

from syl.common import Colors, DockerManager
from syl.common.docker import SYL_IMAGES


def pull_images(docker: DockerManager) -> None:
    """Pull the latest versions of all syl images"""
    log.info('Pulling latest versions of all syl images...')

    results = docker.pull_all_syl_images()

    print()
    print('Pull Results:')
    print('=' * 50)

    success_count = 0
    for image_name, success in results.items():
        if success:
            print(f'{Colors.bright_green("✓")} {image_name}')
            success_count += 1
        else:
            print(f'{Colors.bright_red("✗")} {image_name}')

    print('=' * 50)
    print(f'Successfully pulled {success_count}/{len(results)} images')
    print()

    if success_count == len(results):
        log.info('All images pulled successfully')
    else:
        log.warning(f'Failed to pull {len(results) - success_count} images')
