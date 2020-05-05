"""
original label
"""
ORIGINAL_LABEL = 0


"""
it is used to say we are using the randomforest result label
by directly trained based on real label
"""
PREDICT_BY_LABEL = 1


"""
it is used to say we are using the randomforest result label
by directly trained based on real [x,y] location
"""
PREDICT_BY_COORDINATE = 2


"""
the constant is connected from KNNUtil plot_result, when you want to set the title name equals k
we add this parameter into the plot function at DataUtil . SimpleVisulizeCoord
"""
KNN_BEST_K_RESULT = "best k is "