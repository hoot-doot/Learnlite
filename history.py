import streamlit as st
from firebase_admin import firestore



def app():
    db=firestore.client()


    try:    
        result = db.collection('History').document(st.session_state['useremail']).get()
        r=result.to_dict()
        summary = r['Summary']
        ocr = r['OCR']
        timestamps = r['Timestamp']
        if st.session_state.username=='':
            st.title('History of: '+st.session_state['useremail'] )
        else:
            st.title('History of: '+st.session_state['username'] )
            
        
        def delete_post(k):
            h=ocr[c]
            try:
                db.collection('History').document(st.session_state['useremail']).update({"OCR": firestore.ArrayRemove([h])})
                db.collection('History').document(st.session_state['useremail']).update({"Summary": firestore.ArrayRemove([h])})
                db.collection('History').document(st.session_state['useremail']).update({"Timestamp": firestore.ArrayRemove([h])})
                st.warning('History deleted')
            except:
                st.write('Something went wrong..')
                
        for c in range(len(ocr)):
            st.text_area(label='OCR',value=ocr[c])
            st.text_area(label='Summary',value=summary[c])
            st.text(f'Timestamp: {timestamps[c]}')
            st.button('Delete Post', on_click=delete_post, args=([c] ), key=c)          

        
    except:
        if st.session_state.username=='' and st.session_state.useremail=='':
            st.subheader('Please Login first...')        
