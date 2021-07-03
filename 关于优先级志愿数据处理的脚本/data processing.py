file = open("data.txt", "r")
lines = file.readlines()
file.close()
result = {}
pre_process = []
# result[index] = [first course wish, second, third]
for i in range(14):
    temp = [0, 0, 0]
    result[i + 1] = temp
for line in lines:
    pre_process.append(line.strip().split(" "))
for data_set in pre_process:
    for i in range(len(data_set)):
        result[eval(data_set[i])][i] += 1
file = open("result.txt", "w")
for i in range(14):
    file.write("综%s:"%(str(i+1)))
    file.writelines(str(result[i + 1]))
    file.write("\n")
file.close()

