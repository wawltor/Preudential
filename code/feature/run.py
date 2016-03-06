'''
Created on 2015/12/27

@author: FZY
'''
import os 
if __name__ == '__main__':
    
    cmd = 'python ./BMIFeature.py'
    os.system(cmd)
    cmd = 'python ./FamilyFeature.py'
    os.system(cmd)
    cmd = 'python ./Medical_Keyword.py'
    os.system(cmd)
    #cmd = 'python ./Product_Info2_Feature.py'
    #os.system(cmd)
    cmd = 'python ./MedicalHistoryFeature.py'
    os.system(cmd)
    cmd = 'python ./InsuranceHistoryFeature.py'
    os.system(cmd)
    
    