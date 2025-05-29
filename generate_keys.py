import yaml
import streamlit_authenticator as stauth
 
names = ["login"]
usernames = ["tester"]
passwords = ["Q1w2e3r4"] # yaml 파일 생성하고 비밀번호 지우기!
 
hashed_passwords = stauth.Hasher(passwords).hash_list()
 
data = {
    "credentials" : {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                },
            usernames[1]:{
                "name":names[1],
                "password":hashed_passwords[1]
                }            
            }
    },
    "cookie": {
        "expiry_days" : 1,
        "key": "some_signature_key",
        "name" : "some_cookie_name"
    },
    "preauthorized" : {
        "emails" : [
            "melsby@gmail.com"
        ]
    }
}
 
with open('config.yaml','w') as file:
    yaml.dump(data, file, default_flow_style=False)