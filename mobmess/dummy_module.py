class DummyModule:

    """When a package or module can't be imported, use this as a dummy
    replacement. Getting any attribute from this class will result in
    an error message that the package wasn't imported
    """
    
    def __init__(self, pkg_name, install_cmd=None):
        self.pkg_name = pkg_name

        if install_cmd is not None:
            self.install_cmd = f"Please install with `{self.install_cmd}`"
        else:
            self.install_cmd = ""

    def __getattr__(self, name):
        raise Exception(f"{self.pkg_name} needs to be installed.{self.install_cmd}")
