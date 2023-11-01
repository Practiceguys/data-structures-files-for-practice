print ("Student Information Storing and Printing Program of class")
num_std =int(input("How many Students are in the Class : "))
names , fnames , ages , phones , = [],[],[],[]

for std in range(num_std):
    names.append(input(f"Enter Student # {std+1} name : "))
    fnames.append(input(f"Enter Student # {std+1} father name : "))
    ages.append(int(input(f"Enter Student # {std+1} age : ")))
    phones.append(input(f"Enter Student # {std+1} phone number : "))
    
for std in range(num_std):
    print (f"Information of Mr  {names[std]} ")
    print (f"\t His Father name is {fnames[std]}")
    print (f"\t His age is {ages[std]}")
    print (f"\t His phone number is {phones[std]}")


print (names,fnames,ages,phones)

