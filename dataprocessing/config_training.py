config = {'train_data_path':['../DOWNLOADLUNA16PATH/subset0/',
                             '../DOWNLOADLUNA16PATH/subset1/',
                             '../DOWNLOADLUNA16PATH/subset2/',
                             '../DOWNLOADLUNA16PATH/subset3/',
                             '../DOWNLOADLUNA16PATH/subset4/',
                             '../DOWNLOADLUNA16PATH/subset5/',
                             '../DOWNLOADLUNA16PATH/subset6/',
                             '../DOWNLOADLUNA16PATH/subset7/',
                             '../DOWNLOADLUNA16PATH/subset8/'],
          'val_data_path':['../DOWNLOADLUNA16PATH/subset9/'], 
          'test_data_path':['../DOWNLOADLUNA16PATH/subset9/'], 
          
          'train_preprocess_result_path':'../LUNA16PROPOCESSPATH/', 
          'val_preprocess_result_path':'../LUNA16PROPOCESSPATH/',  
          'test_preprocess_result_path':'../LUNA16PROPOCESSPATH/',
          
          'train_annos_path':'../LUNA16ANNOTATIONPATH/luna16/CSVFILES/annotations.csv',
          'val_annos_path':'../LUNA16ANNOTATIONPATH/luna16/CSVFILES/annotations.csv',
          'test_annos_path':'../LUNA16ANNOTATIONPATH/luna16/CSVFILES/annotations.csv',

          'black_list':[],
          
          'preprocessing_backend':'python',

          'luna_segment':'../LUNA16SEGMENTATIONPATH/luna16/seg-lungs-LUNA16/', # download from https://luna16.grand-challenge.org/data/
          'preprocess_result_path':'../LUNA16PROPOCESSPATH/',
          'luna_data':'../DOWNLOADLUNA16PATH/',
          'luna_label':'../LUNA16ANNOTATIONPATH/luna16/CSVFILES/annotations.csv'
         } 
