counter = 0
sum = 0.0
sparity = 0.05
targets = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
memory_size = 12288 # in flaots minus warp reduction

for target in targets:
    while True:
        if sum >= target:
            print("target: ", target)
            print("counter: ", counter)
            print("modulo 32: ", counter % 32)
            print("fit remoddel: ", (counter + (32 - (counter % 32))) * (counter - (counter % 32)) * 2 <= memory_size)
            print("fit remoddel size: ", (counter + (32 - (counter % 32))) * (counter - (counter % 32)))
            print("fit remoddel dim: ", (counter + (32 - (counter % 32))), "x", (counter - (counter % 32)))
            print("fit remoddel diff: ", abs(memory_size - 2 * ((counter + (32 - (counter % 32))) * (counter - (counter % 32)))))
            print("----------------------------")
            break  
        sum += sparity
        counter += 1
    sum = 0.0
    counter = 0
