import argparse
import datetime
from integration import TBRGSIntegration


def main():
    parser = argparse.ArgumentParser(description='TBRGS - Hệ thống tìm đường dựa trên dự đoán lưu lượng giao thông.')

    parser.add_argument('--problem', required=True, help='Đường dẫn đến file chứa thông tin đồ thị')
    parser.add_argument('--method', default='AS', choices=['DFS', 'BFS', 'GBFS', 'AS', 'CUS1', 'CUS2'],
                        help='Phương pháp tìm đường')
    parser.add_argument('--date', help='Ngày di chuyển (format: YYYY-MM-DD)')
    parser.add_argument('--time', help='Thời gian di chuyển (format: HH:MM)')

    args = parser.parse_args()

    # Xử lý thời gian di chuyển
    if args.date and args.time:
        date_parts = list(map(int, args.date.split('-')))
        time_parts = list(map(int, args.time.split(':')))
        travel_time = datetime.datetime(date_parts[0], date_parts[1], date_parts[2],
                                        time_parts[0], time_parts[1])
    else:
        travel_time = datetime.datetime.now()

    # Khởi tạo hệ thống tích hợp
    system = TBRGSIntegration(search_method=args.method)

    # Tìm đường đi tối ưu
    goal, nodes_created, path = system.find_optimal_route(args.problem, travel_time)

    # In kết quả
    print(f"Đường đi tối ưu từ node gốc đến node {goal}:")
    print(f"Số node đã khám phá: {nodes_created}")
    print(f"Đường đi: {' -> '.join(map(str, path))}")

    # Tính tổng thời gian di chuyển
    if path:
        with open(args.problem, 'r') as f:
            contents = f.read()

        # Dùng hàm parse_problem_file để phân tích file
        import sys
        sys.path.append('../part_a')
        from search import parse_problem_file
        nodes, edges, origin, destinations = parse_problem_file(args.problem)

        travel_time = system.get_travel_time(path, nodes, travel_time)
        print(f"Thời gian di chuyển ước tính: {travel_time:.2f} giờ ({travel_time * 60:.0f} phút)")


if __name__ == "__main__":
    main()