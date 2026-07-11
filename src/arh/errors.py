class InfraError(Exception):
    """A failure caused by infrastructure (Docker, MCP transport, provider 5xx/429),
    NOT by the agent. Infra failures are retried and never counted against pass^k."""
