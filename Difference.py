import streamlit as st

# Function to find differences between two text files
def find_differences(file1, file2, output_file):
    text1 = file1.readlines()
    text2 = file2.readlines()

    # Find lines that are unique to each file
    unique_to_file1 = [line for line in text1 if line not in text2]
    unique_to_file2 = [line for line in text2 if line not in text1]

    # Write the differences to the output file
    with open(output_file, 'w') as output:
        output.write("Lines unique to the first file:\n")
        output.writelines(unique_to_file1)
        output.write("\nLines unique to the second file:\n")
        output.writelines(unique_to_file2)

def main():
    st.title("Text File Comparison App")

    # Upload the first text file
    uploaded_file1 = st.file_uploader("Upload the first text file", type=["txt"])
    if uploaded_file1 is not None:
        text1 = uploaded_file1

    # Upload the second text file
    uploaded_file2 = st.file_uploader("Upload the second text file", type=["txt"])
    if uploaded_file2 is not None:
        text2 = uploaded_file2

    # Name for the output file
    output_file_name = st.text_input("Enter the output file name (e.g., differences.txt)")

    # Compare the files when a button is clicked
    if st.button("Compare Files"):
        if uploaded_file1 is not None and uploaded_file2 is not None and output_file_name:
            find_differences(text1, text2, output_file_name)
            st.write("Differences have been saved to", output_file_name)

if __name__ == "__main__":
    main()
