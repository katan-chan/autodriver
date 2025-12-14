from pathlib import Path
import osmnx as ox


# ==== Cấu hình ====
OUTPUT_PATH = Path("data/hanoi_roads.graphml")

# Bộ lọc "rộng" — giữ gần như toàn bộ đường xe máy có thể đi
# - Lấy tất cả các đường có tag highway
# - Loại bỏ các loại đường rõ ràng chỉ đi bộ: footway, path, steps, corridor, elevator, escalator, pedestrian
# - Giữ lại trừ khi có tag "motor_vehicle=no" hoặc "access=no" hoặc "motorcycle=no"
CUSTOM_FILTER = (
    '["highway"]'
    '["highway"!~"footway|path|steps|corridor|elevator|escalator|pedestrian"]'
    '["motor_vehicle"!~"no"]["access"!~"no"]["motorcycle"!~"no"]'
)


def download_hanoi_motorbike_graph(output_path: Path) -> Path:
    """
    Download road network of Hanoi suitable for motorbikes (including small alleys)
    and save as GraphML file.

    Args:
        output_path: file path to save the GraphML output
    Returns:
        Path to the saved GraphML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("🚀 Đang tải đồ thị đường Hà Nội (dành cho xe máy, gồm cả ngõ/ngách)...")
    G = ox.graph_from_place(
        "Hanoi, Vietnam",
        network_type="all",        # không giới hạn loại mạng (để không bị mất hẻm nhỏ)
        custom_filter=CUSTOM_FILTER,
        retain_all=True,           # giữ mọi thành phần (ngõ nhỏ, khu riêng lẻ)
        truncate_by_edge=True,     # giữ cả cạnh cắt biên
        simplify=True,             # gộp node bậc 2 để giảm kích thước
    )

    print(f"💾 Lưu đồ thị vào {output_path} ...")
    ox.save_graphml(G, output_path)
    print("✅ Hoàn tất tải và lưu đồ thị xe máy Hà Nội.")
    print(f"→ Tổng số nút: {len(G.nodes):,}, cạnh: {len(G.edges):,}")
    return output_path


if __name__ == "__main__":
    download_hanoi_motorbike_graph(OUTPUT_PATH)
