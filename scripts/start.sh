#!/bin/bash

cp /home/ec2-user/arpo/.env_prod /home/ec2-user/arpo/.env 
source /home/ec2-user/arpo/bin/activate
nohup /home/ec2-user/arpo/bin/streamlit run /home/ec2-user/arpo/streamlit.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false 1> /home/ec2-user/arpo/logs/output.log 2>&1 &