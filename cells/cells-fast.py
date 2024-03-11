def cells(passes: int, cells: int) -> list[bool]:
    for _ in range(1, passes+1):
        cells_list = list([False if _ == 1 else not cells_list[i] if (i+1) % _ == 0 else cells_list[i] for i in range(0, cells)])
    return cells_list


print(cells(500, 500))
