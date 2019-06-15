seriers = 0
count = 0
def make_averager():
    def average(new_value):
        global seriers, count
        seriers += new_value
        count += 1
        return seriers / count

    return average

avg = make_averager()
print(avg(10))
print(avg(11))
print(avg(12))

