import os

def prepare_dataset():
    """
        This function prepares the dataset for training: moving all the videos to a single directory because that is how the question answer annotations are given.
    """
    rootdir = 'activityNet/'
    DEST_PATH = 'activityNet/all_videos/'

    print("Root Dir : ",rootdir)

    try :
        if not os.path.exists('activityNet/all_videos'):
            os.makedirs('activityNet/all_videos')

    except OSError:
        print('Error: Creating directory of data')

    for subdir, dirs, files in os.walk(rootdir):
        for parent in dirs:
            for subdir,dir, files in os.walk(rootdir+parent):
                print("Files : ",subdir)
                #move all the files to all_videos
                for file in files:
                    print("From : ",rootdir+parent+'/'+file,"\n")
                    os.rename(rootdir+parent+'/'+file,DEST_PATH+file)


if __name__ == "__main__":
    prepare_dataset() 