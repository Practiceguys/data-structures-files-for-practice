class Pq:
    def __init__(self,size):
        self.size = size
        self.que = list(range(size))
        self.nitems = 0
    def insert(self,item):
        j =0
        if self.nitems == 0:
            self.que.insert(0,item)
            self.nitems+=1
#             print (self.nitems)
        else:
            j = self.nitems-1
            while (j >= 0):
                if (item > self.que[j]):
                    self.que[j+1] = self.que[j]
                else:
                    break
                j-=1
            self.que[j+1] = item
            self.nitems += 1
    def remove(self):
        self.nitems -=1
        return self.que[self.nitems]
    def peekMin(self):
        return self.que[self.nitems-1]
    def isEmpty(self):
        return self.nitems == 0
    def isFull(self):
        return self.nitems == self.size

obj = Pq(5)
obj.insert(30)
obj.insert(50)
obj.insert(10)
obj.insert(40)
obj.insert(20)
while not(obj.isEmpty()):
    print(obj.remove())
