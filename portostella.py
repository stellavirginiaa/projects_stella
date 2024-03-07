import streamlit as st
import streamlitphishing

st.write("""
<style>
.logo-container {
    display: flex;
    align-items: center;
    gap: 20px;
}

.logo-container img {
    height: 30px;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title('Portofolio Project Stella')

    st.markdown("""
    <div class="logo-container">
    <div class="logo">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png">
        <p>@stellavirginia</p>
    </div>
    <div class="logo">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png">
        <p>stellavirginia02@gmail.com</p>
    </div>
    <div class="logo">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png">
        <p><a href="https://www.linkedin.com/in/stellavirginia">Stella Virginia</a></p>
    </div>
    <div class="logo">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg">
        <p><a href="https://github.com/stellavirginiaa">stellavirginiaa</a></p>
    </div>
    </div>  
    """, unsafe_allow_html=True)

    st.write("""
    ## About Me
    """)

    with st.expander("Show Profile Description"):
        st.write("""
        Hello there! I'm Stella Virginia, a passionate and driven individual with a strong background in Informatics, specializing in Artificial Intelligence. I'm currently pursuing my undergraduate degree at Universitas Bunda Mulia.

        My fascination lies in the realm of data analysis and machine learning. I thrive on the challenges presented by complex datasets and am always eager to explore innovative solutions using cutting-edge technologies. Throughout my academic journey, I've actively engaged in various research projects, where I've had the opportunity to apply my skills in AI to real-world problems. I believe that the power of data-driven insights can revolutionize industries and drive positive change in society.

        Beyond academics, I'm an avid learner and enjoy keeping up-to-date with the latest advancements in technology. I'm also a firm believer in collaboration and am enthusiastic about working with like-minded individuals to tackle exciting challenges and make a meaningful impact. If you'd like to connect or discuss potential opportunities, feel free to reach out to me via [email](mailto:stellavirginia02@gmail.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/stellavirginia). I look forward to connecting with you!
        """)
    

    st.write("""
    ## Projects Demo
    """)

    project_options = ["Phishing Link Detection", "", ""]
    selected_project = st.selectbox("Projects:", project_options)

    if selected_project == "Phishing Link Detection":
        st.write("""
        ### Phishing Link Detection
        """)
        streamlitphishing.main()
    

if __name__ == "__main__":
    main()
