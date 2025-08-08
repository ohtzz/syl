class Colors:
    """ANSI color codes for terminal output"""

    # Basic colors (4-bit)
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors (4-bit)
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # 8-bit colors (256 color palette)
    ORANGE = '\033[38;5;208m'
    PURPLE = '\033[38;5;129m'
    PINK = '\033[38;5;205m'
    LIME = '\033[38;5;154m'
    TEAL = '\033[38;5;45m'
    NAVY = '\033[38;5;17m'
    MAROON = '\033[38;5;88m'
    OLIVE = '\033[38;5;100m'
    AQUA = '\033[38;5;51m'
    FUCHSIA = '\033[38;5;201m'
    SILVER = '\033[38;5;248m'
    GRAY = '\033[38;5;244m'
    INDIGO = '\033[38;5;54m'
    CORAL = '\033[38;5;203m'
    GOLD = '\033[38;5;220m'
    CRIMSON = '\033[38;5;196m'
    LAVENDER = '\033[38;5;183m'
    TURQUOISE = '\033[38;5;80m'
    SALMON = '\033[38;5;209m'
    VIOLET = '\033[38;5;177m'

    # Text formatting
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'

    @staticmethod
    def black(text):
        return f'{Colors.BLACK}{text}{Colors.RESET}'

    @staticmethod
    def red(text):
        return f'{Colors.RED}{text}{Colors.RESET}'

    @staticmethod
    def green(text):
        return f'{Colors.GREEN}{text}{Colors.RESET}'

    @staticmethod
    def yellow(text):
        return f'{Colors.YELLOW}{text}{Colors.RESET}'

    @staticmethod
    def blue(text):
        return f'{Colors.BLUE}{text}{Colors.RESET}'

    @staticmethod
    def magenta(text):
        return f'{Colors.MAGENTA}{text}{Colors.RESET}'

    @staticmethod
    def cyan(text):
        return f'{Colors.CYAN}{text}{Colors.RESET}'

    @staticmethod
    def white(text):
        return f'{Colors.WHITE}{text}{Colors.RESET}'

    @staticmethod
    def bright_black(text):
        return f'{Colors.BRIGHT_BLACK}{text}{Colors.RESET}'

    @staticmethod
    def bright_red(text):
        return f'{Colors.BRIGHT_RED}{text}{Colors.RESET}'

    @staticmethod
    def bright_green(text):
        return f'{Colors.BRIGHT_GREEN}{text}{Colors.RESET}'

    @staticmethod
    def bright_yellow(text):
        return f'{Colors.BRIGHT_YELLOW}{text}{Colors.RESET}'

    @staticmethod
    def bright_blue(text):
        return f'{Colors.BRIGHT_BLUE}{text}{Colors.RESET}'

    @staticmethod
    def bright_magenta(text):
        return f'{Colors.BRIGHT_MAGENTA}{text}{Colors.RESET}'

    @staticmethod
    def bright_cyan(text):
        return f'{Colors.BRIGHT_CYAN}{text}{Colors.RESET}'

    @staticmethod
    def bright_white(text):
        return f'{Colors.BRIGHT_WHITE}{text}{Colors.RESET}'

    @staticmethod
    def orange(text):
        return f'{Colors.ORANGE}{text}{Colors.RESET}'

    @staticmethod
    def purple(text):
        return f'{Colors.PURPLE}{text}{Colors.RESET}'

    @staticmethod
    def pink(text):
        return f'{Colors.PINK}{text}{Colors.RESET}'

    @staticmethod
    def lime(text):
        return f'{Colors.LIME}{text}{Colors.RESET}'

    @staticmethod
    def teal(text):
        return f'{Colors.TEAL}{text}{Colors.RESET}'

    @staticmethod
    def navy(text):
        return f'{Colors.NAVY}{text}{Colors.RESET}'

    @staticmethod
    def maroon(text):
        return f'{Colors.MAROON}{text}{Colors.RESET}'

    @staticmethod
    def olive(text):
        return f'{Colors.OLIVE}{text}{Colors.RESET}'

    @staticmethod
    def aqua(text):
        return f'{Colors.AQUA}{text}{Colors.RESET}'

    @staticmethod
    def fuchsia(text):
        return f'{Colors.FUCHSIA}{text}{Colors.RESET}'

    @staticmethod
    def silver(text):
        return f'{Colors.SILVER}{text}{Colors.RESET}'

    @staticmethod
    def gray(text):
        return f'{Colors.GRAY}{text}{Colors.RESET}'

    @staticmethod
    def indigo(text):
        return f'{Colors.INDIGO}{text}{Colors.RESET}'

    @staticmethod
    def coral(text):
        return f'{Colors.CORAL}{text}{Colors.RESET}'

    @staticmethod
    def gold(text):
        return f'{Colors.GOLD}{text}{Colors.RESET}'

    @staticmethod
    def crimson(text):
        return f'{Colors.CRIMSON}{text}{Colors.RESET}'

    @staticmethod
    def lavender(text):
        return f'{Colors.LAVENDER}{text}{Colors.RESET}'

    @staticmethod
    def turquoise(text):
        return f'{Colors.TURQUOISE}{text}{Colors.RESET}'

    @staticmethod
    def salmon(text):
        return f'{Colors.SALMON}{text}{Colors.RESET}'

    @staticmethod
    def violet(text):
        return f'{Colors.VIOLET}{text}{Colors.RESET}'

    @staticmethod
    def bold(text):
        return f'{Colors.BOLD}{text}{Colors.RESET}'

    @staticmethod
    def dim(text):
        return f'{Colors.DIM}{text}{Colors.RESET}'

    @staticmethod
    def italic(text):
        return f'{Colors.ITALIC}{text}{Colors.RESET}'

    @staticmethod
    def underline(text):
        return f'{Colors.UNDERLINE}{text}{Colors.RESET}'

    @staticmethod
    def strikethrough(text):
        return f'{Colors.STRIKETHROUGH}{text}{Colors.RESET}'

    @staticmethod
    def custom_256(color_code, text):
        """Use any 256-color palette color (0-255)"""
        return f'\033[38;5;{color_code}m{text}{Colors.RESET}'

    @staticmethod
    def custom_rgb(r, g, b, text):
        """Use RGB colors (0-255)"""
        return f'\033[38;2;{r};{g};{b}m{text}{Colors.RESET}'
