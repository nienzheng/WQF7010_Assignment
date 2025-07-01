# class GlobalProperties:
#     black_threshold = 120
#     line_votes = 150
#     line_space_threshold = 40
#     min_cross_angle = 0.55
#     max_cross_angle = 1.05
#     histeresis = 30
class GlobalProperties:
    black_threshold = 80              # ↓ 降低阈值，保留更多黑色线条（防止误白）
    line_votes = 80                  # ↓ 降低投票数，容许模糊线也算作直线
    line_space_threshold = 50        # ↑ 放宽线间隔合并容忍度
    min_cross_angle = 0.45           # 保持 X 容错能力
    max_cross_angle = 1.10
    histeresis = 30
