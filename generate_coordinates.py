import random

# ===== 配置参数 =====
NUM_LINES = 50  # 生成的行数，可以修改这个值
GRID_SIZE = 20  # 网格大小 (20x20)，注意：坐标范围是 0 到 GRID_SIZE-1，即 0-19
OUTPUT_FILE = "game_coordinates.txt"  # 输出文件名
# ===================

def generate_coordinates():
    """生成一组不重复的坐标 (3个猎人 + 1个猎物)
    坐标范围: 0 到 GRID_SIZE-1 (即 0-19)
    """
    # range(GRID_SIZE) 生成 0 到 GRID_SIZE-1，即 0-19
    all_positions = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
    selected = random.sample(all_positions, 4)
    return selected

def format_line(coords):
    """将坐标列表格式化为字符串"""
    return " ".join([f"({x}, {y})" for x, y in coords])

def main():
    print(f"开始生成 {NUM_LINES} 行坐标数据...")
    print(f"网格大小: {GRID_SIZE}x{GRID_SIZE}")
    print(f"坐标范围: 0 到 {GRID_SIZE-1} (即 0-19)")
    print(f"输出文件: {OUTPUT_FILE}")
    print("=" * 50)
    
    all_coords = []
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i in range(NUM_LINES):
            coords = generate_coordinates()
            all_coords.extend(coords)  # 收集所有坐标用于验证
            line = format_line(coords)
            f.write(line + '\n')
            
            # 每10行打印一次进度
            if (i + 1) % 10 == 0:
                print(f"已生成 {i + 1}/{NUM_LINES} 行")
    
    # 验证坐标范围
    max_x = max(coord[0] for coord in all_coords)
    max_y = max(coord[1] for coord in all_coords)
    min_x = min(coord[0] for coord in all_coords)
    min_y = min(coord[1] for coord in all_coords)
    
    print("=" * 50)
    print(f"✅ 成功生成 {NUM_LINES} 行坐标数据到 {OUTPUT_FILE}")
    print(f"每行格式: (x1, y1) (x2, y2) (x3, y3) (x4, y4)")
    print(f"前3个坐标为猎人位置，第4个坐标为猎物位置")
    print(f"\n坐标验证:")
    print(f"  X范围: {min_x} 到 {max_x}")
    print(f"  Y范围: {min_y} 到 {max_y}")
    print(f"  ✓ 所有坐标都在 0-{GRID_SIZE-1} 范围内" if max_x < GRID_SIZE and max_y < GRID_SIZE else "  ✗ 坐标超出范围！")

if __name__ == "__main__":
    main()
