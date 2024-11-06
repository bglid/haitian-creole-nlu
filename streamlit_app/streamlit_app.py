#Front end application, for the users!
import streamlit as st

def main():
    #Test example
    st.title('Haitian Creole Automatic Grader')
    st.header('*A tool for assisting in teaching Haitian Creole*', divider=True)
    st.header('Question 1', divider=True)

    #Dictionary to hold the questions and responses
    # question_dict = {} #key=question : val=response

    # #Set to hold that a question has been answered
    question_set = set()

    question_1 = st.text_input('Please describe your morning routine in Haitian Creole', key='q1')
    if question_1:
        submit_1 = st.button('Submit answer to question 1')
        if submit_1:
            st.write('Answer submitted')
            st.balloons()
            #Adding to set of questions answered
            # question_dict['q1'] = question_1
            question_set.add('q1')

    st.header('Question 2', divider=True)
    question_2_text = st.write('Test question...')
    question_2 = st.selectbox('Please select the correct answer to question 2:',
                            options=[
                                'Answer 1',
                                'Answer 2', 
                                'Answer 3'
                            ], key='q2')

    submit_2 = st.button('Submit answer to question 2')
    try:
        if submit_2 and question_2:
            st.write('Answer submitted')
            st.balloons()
            # question_dict['q2'] = question_2
            question_set.add('q2')
        else:
            st.write('please select an answer for question 2...')
            
    except Exception as e:
        st.write(e)
    
    st.session_state.keys()
    
    
    #Setting progress bar for questions completed:
    progress_text = 'Percentage of questions completed'
    progress_bar = st.progress(len(question_set) / 2, progress_text) #need to update to be scalable

if __name__ == "__main__":
    main()
    
