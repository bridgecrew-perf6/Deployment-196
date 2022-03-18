In the setup.sh file we will create a streamlit folder with a credentials.toml and a config.toml file.

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"immangeek@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml