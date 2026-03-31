"""Link class for network links with delay and bandwidth."""


class Link:
    """网络链路"""

    def __init__(self, src: str, dst: str, delay: float, bandwidth: float = 1000.0):
        """
        Initialize a network link.

        Args:
            src: Source node ID
            dst: Destination node ID
            delay: Propagation delay in ms
            bandwidth: Bandwidth in Mbps
        """
        self.src = src
        self.dst = dst
        self.delay = delay
        self.bandwidth = bandwidth

    def __repr__(self) -> str:
        return f"Link(src='{self.src}', dst='{self.dst}', delay={self.delay}, bandwidth={self.bandwidth})"
