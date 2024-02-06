import tabulate
def cells(passes, n):
    cells = [{i: False for i in range(1,n+1)}]
    for i in range(0, passes):
        cells.append(dict())
        for cell in cells[i]:
            if cell % (i+2) == 0:
                if cells[i][cell] == False:
                    cells[i+1][cell] = True
                else:
                    cells[i+1][cell] = False
            else:
                cells[i+1][cell] = cells[i][cell]
    return cells


cells_list = cells(3, 7)
header = cells_list[0].keys()
rows = [x.values() for x in cells_list]
print(tabulate.tabulate(rows, header, tablefmt="simple"))