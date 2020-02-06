import os

if __name__ == '__main__':
#    dataset_list=['Pathway1_1']
    dataset_list=['Pathway2_1','Pathway2_2','Pathway2_3','Doorway_1','Doorway_2','Doorway_3','Room_2'
        ,'Staircase_1','Staircase_2','Staircase_3','Staircase_4','Bus_2']

#    dataset_list = ['Pathway2_1','Doorway_3','Room_2','Staircase_2','Staircase_3','Staircase_4']
#    output_name = 'mobilenet_v2_coco_20180329'
    #output_name=ssd_inception_v2_coco_2017_11_17
    for dt in dataset_list:
        print(dt)
        os.system("python SSD.py -i /work/superorange5/MI3 -d "+dt+" -l -m 3")
    


