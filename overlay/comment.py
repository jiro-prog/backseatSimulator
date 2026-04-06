import time
from dataclasses import dataclass, field


@dataclass
class Comment:
    text: str
    color: str
    y_pos: int
    x_pos: float
    speed: float
    font_size: int
    slot: int = -1
    text_width: int = 0
    created_at: float = field(default_factory=time.time)

    def is_offscreen(self) -> bool:
        """画面左端を超えたか判定（テキスト幅分の余裕を持たせる）"""
        return self.x_pos < -(len(self.text) * self.font_size)

    def update(self):
        """x_posをspeed分だけ減算"""
        self.x_pos -= self.speed
