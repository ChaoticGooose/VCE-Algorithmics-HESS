import csv
def cells(passes, n):
    cells = [{i: False for i in range(1,n+1)}] # Set every cell to unlocked initally
    for i in range(0, passes-1): 
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

# CSV output writer
cells_list = cells(500, 500)
header = cells_list[0].keys()
with open('cells.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, header)
    dict_writer.writeheader()
    dict_writer.writerows(cells_list)
print("csv written to ./cells.csv")