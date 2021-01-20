# cmpe537-image-classification
Local Descriptor based Image Classification

pipeline_mansur.py contains all HOG based pipelines.  
To run pipeline_mansur.py, directory structure should be as follows:

- src
  - pipeline_mansur.py  
  - bag_of_words.py  
  - image_descriptor.py  
  - spectral_clustering.py  
  - train_classifier.py  
- Caltech20  
  - training  
  - testing  


For the ORB - Hierachical K-Means - Bag of Visual Words - SVM pipeline

    run bovw_pipeline.py after setting the path of the dataset.
    If you want to recalculate K-means clusters again
    run hier_kmeans.py after setting the path of the dataset and K.


For the ORB - GMM - Fisher Vectors - SVM pipeline

    run gmm_fisher_pipeline.py after setting the path of the dataset.
