import os
def delete_num(srcpath,outpath):
    filelist=os.listdir(srcpath)
    os.makedirs(outpath)
    for file in filelist:
        f=open(srcpath+'/'+file)
        scr_name=file.split('.')[0]
        dst_name=scr_name+'.txt'
        outfile=open(outpath+'/'+dst_name,'w')
        for line in f.readlines():
            line=line[:-1]
            line=line.rstrip()
            outfile.write(line+'\n')
    f.close()
    outfile.close()
if __name__=='__main__':
    delete_num('/home/suncheng/yolov5_facedetection/inference/output','/home/suncheng/yolov5_facedetection/inference/newfile')