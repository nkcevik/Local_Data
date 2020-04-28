Memory Efficient Learning

Expect 95-97% of training time to be spent fitting the base estimators. 
Training time depends primarily on the number of base learners in the ensemble, 
the number of threads or cores available, and the size of the dataset. Speaking 
of size, ensembles that partition the data during training scale more efficiently
than their base learners.

Note that:	score_m = score of root mean square error (rmse), 
		score_s = score of standard deviation (std),
           	ft_m = fit time of model for rmse, 
		ft_s = fit time of model for std,
           	pt_m = prediction time of model for rmse, 
		pt_s = prediction time of model for std

For more information, see the link or the Documents folder:
https://machinelearningmastery.com/super-learner-ensemble-in-python/
https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html

