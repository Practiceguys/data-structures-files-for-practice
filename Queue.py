class Pq:
    def __init__(self,size):
        self.size = size
        self.que = list(range(size))
        self.nitems = 0
        self.first = 0
        self.last = -1
    def insert(self,item):
        if self.last == self.size -1:
            self.last -=1
        self.last+=1
        self.que[self.last] = item
        self.nitems+=1
    def remove(self):
        temp = self.que[self.first]
        self.first+=1
        if self.first == self.size+1:
            first = 0
        self.nitems-=1
        return temp
    def peekFirst(self):
        return self.que[self.first]
    def isEmpty(self):
        return self.nitems == 0
    def isFull(self): 
        return self.nitems == self.size
    def qSize(self): 
        return self.nitems

obj = Pq(5); obj.insert(30); obj.insert(50); obj.insert(10); obj.insert(40); obj.insert(20)
print (obj.que)
while not(obj.isEmpty()):
    print(obj.remove())

