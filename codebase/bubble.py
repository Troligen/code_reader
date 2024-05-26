class BubbleSorter():
    def __init__(self, list):
        self.list = list

    def sort(self):
        for _ in range( len( self.list ) ):
            did_swap = False
            for j in range(len( self.list )):
                if j + 1 == len(self.list):
                    continue
                elif self.list[j] > self.list[j + 1]:
                    swap = self.list[j]
                    self.list[j] = self.list[j + 1]
                    self.list[j + 1] = swap
                    did_swap = True
            if not did_swap:
                break
    def get_storted_list(self):
        self.sort()
        return self.list

sort_this = BubbleSorter([5,1,2,1,10,5,2000,7,53,5435,2,7,767,3,5,7453,7,33453,1])

print(sort_this.get_storted_list())