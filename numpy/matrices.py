import numpy

#creating matrix
>>> a = numpy.matrix('5,7,4;2,3,6;7,8,9')
>>> a
matrix([[5, 7, 4],
        [2, 3, 6],
        [7, 8, 9]])

>>> b = numpy.matrix('1,2;3,4;5,6')
>>> 
>>> b
matrix([[1, 2],
        [3, 4],
        [5, 6]])


#Transpose
>>> c = b.transpose()
>>> c
matrix([[1, 3, 5],
        [2, 4, 6]])


##Multiplication
>>> d = a*b
>>> d
matrix([[ 46,  62],
        [ 41,  52],
        [ 76, 100]])

##Shape
>>> a.shape
(3, 3)
>>> b.shape
(3, 2)


##Inverse
>>> a.I
matrix([[-0.48837209, -0.72093023,  0.69767442],
        [ 0.55813953,  0.39534884, -0.51162791],
        [-0.11627907,  0.20930233,  0.02325581]])
