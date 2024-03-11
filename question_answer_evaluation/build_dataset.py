import os
import json
from collections import defaultdict


'''
Have all the dataset in a directory named train_dataset.
'''
dataset_path = '../train.json'


def default_value():
    return []



class Question():
    def __init__(self,question_id,question,question_type,answer,is_choice_type = False,choices={}):
        self.question_id = question_id
        self.question = question
        self.question_type = question_type
        self.choices = choices
        self.answer = answer
        self.is_choice_type = is_choice_type
    def display_question(self):
        print("Question : ",self.question)
        print("Question Type : ",self.question_type)
        if(self.is_choice_type):
            for key in self.choices.keys():
                print("Choice ",key," : ",self.choices[key])
        
        print("Answer : ",self.answer)

question_database = defaultdict(default_value)

def prepare_dataset():
    """
        This function prepares the dataset for training: moving all the videos to a single directory because that is how the question answer annotations are given.
    """
    rootdir = '../target/'

    try :
        if not os.path.exists('../all_videos'):
            os.makedirs('../all_videos')

    except OSError:
        print('Error: Creating directory of data')

    for subdir, dirs, files in os.walk(rootdir):
        for parent in dirs:
            print("Dir : ",parent,"\n")
            for subdir,dir, files in os.walk(rootdir+parent):
                print("Files : ",files)
                #move all the files to all_videos
                for file in files:
                    os.rename(rootdir+parent+'/'+file,'../all_videos/'+file)
    
def read_annotation():
    dataset_json = json.load(open(dataset_path))
    print(len(dataset_json))
    QUESTION_KEY = "question"
    CHOICES_KEY = "choices"
    for item in dataset_json:
        scene_index = item['scene_index']
        video_filename = item['video_filename']
        #retrieve the integer after "sim_" in this string
        video_filename = video_filename.split('_')[1]
        full_video_path = 'all_videos/'+video_filename
        question_list = item['questions']
        
        for question in question_list:
            if QUESTION_KEY in question and CHOICES_KEY in question:
                question_str = question["question"]
                question_id = question["question_id"]
                question_type = question["question_type"]
                choices = {}
                choice_idx = 0
                correct_choice_idx = -1
                for choice in question["choices"]:
                    choice_idx = choice["choice_id"]
                    answer_status = choice["answer"]
                    if(answer_status == "correct"):
                        correct_choice_idx = choice_idx
                    choices[choice_idx] = choice["choice"]
                question_object = Question(question_id=question_id,question=question_str,question_type=question_type,answer=correct_choice_idx,is_choice_type=True,choices=choices)
                question_database[video_filename].append(question_object)

            else:
                question_str = question["question"]
                question_id = question["question_id"]
                question_family = question["question_family"]
                answer = question["answer"]
                question_object = Question(question_id=question_id,question=question_str,question_type=question_family,is_choice_type=False,answer=answer)
                question_database[video_filename].append(question_object)

             



if __name__ == "__main__":
    '''
    The prepare dataset is used to move all the videos to a single directory.
    Comment  after running once.
    '''
    prepare_dataset()
    '''
    read the question answer json file and store in query-able structure for ease of understanding.
    '''
    read_annotation()
    #Use whatever video you want to test
    video_list = []
    for video in question_database.keys():
        video_list.append(video)
    sample_list = video_list[:3]
    for video in sample_list:
        print("***"*20)
        print("Video : ",video,"\n")
        for questions in question_database[video]:
            questions.display_question()
            print("***********************\n\n")