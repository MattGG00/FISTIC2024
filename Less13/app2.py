import streamlit as st
import pandas as pd
import xlsxwriter
import io


def main():
    st.title("Data Transformation")
    uploaded_file = st.file_uploader("Choose a file",type={"xlsx"})
    if uploaded_file is not None:
        ###### transformation #####################################
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)


        if st.button('Start Processing', help="Process Dataframe"):
            st.header('Addes Column')
            df['new_col'] = 1
            st.dataframe(df)
            st.balloons()
        
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
                df.to_excel(writer, index=False)
                processed_data = buffer.getvalue()
                st.download_button(
                                    label="Scarica il file processato",
                                    data=processed_data,
                                    file_name="Ordinante2.xlsx",
                                    mime="application/vnd.ms-excel"
                                    )
                
if __name__ == "__main__":
    main()