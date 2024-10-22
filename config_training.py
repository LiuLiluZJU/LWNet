config = {'train_data_path':['../LUNA16/DOWNLOADLUNA16PATH/subset9/',
                             '../LUNA16/DOWNLOADLUNA16PATH/subset1/',
                             '../LUNA16/DOWNLOADLUNA16PATH/subset2/',
                             '../LUNA16/DOWNLOADLUNA16PATH/subset3/',
                             '../LUNA16/DOWNLOADLUNA16PATH/subset4/',
                             '../LUNA16/DOWNLOADLUNA16PATH/subset5/',
                             '../LUNA16/DOWNLOADLUNA16PATH/subset6/',
                             '../LUNA16/DOWNLOADLUNA16PATH/subset7/',
                             '../LUNA16/DOWNLOADLUNA16PATH/subset8/'],
          'val_data_path':['../LUNA16/DOWNLOADLUNA16PATH/subset0/'], 
          'test_data_path':['../LUNA16/DOWNLOADLUNA16PATH/subset0/'], 
          
          'train_preprocess_result_path':'../LUNA16/LUNA16PROPOCESSPATH/', 
          'val_preprocess_result_path':'../LUNA16/LUNA16PROPOCESSPATH/',  
          'test_preprocess_result_path':'../LUNA16/LUNA16PROPOCESSPATH/',
          
          'train_annos_path':'../LUNA16/LUNA16ANNOTATIONPATH/luna16/CSVFILES/annotations.csv',
          'val_annos_path':'../LUNA16/LUNA16ANNOTATIONPATH/luna16/CSVFILES/annotations.csv',
          'test_annos_path':'../LUNA16/LUNA16ANNOTATIONPATH/luna16/CSVFILES/annotations.csv',

          'black_list':[],
          
          'preprocessing_backend':'python',

          'luna_segment':'../LUNA16/LUNA16SEGMENTATIONPATH/luna16/seg-lungs-LUNA16/', # download from https://luna16.grand-challenge.org/data/
          'preprocess_result_path':'../LUNA16/LUNA16PROPOCESSPATH/',
          'luna_data':'../LUNA16/DOWNLOADLUNA16PATH/',
          'luna_label':'../LUNA16/LUNA16ANNOTATIONPATH/luna16/CSVFILES/annotations.csv'
         } 
