from dataclasses import dataclass


@dataclass
class DiscordServer:
    guild_id: str
    channels: list
    self_esteem: int
