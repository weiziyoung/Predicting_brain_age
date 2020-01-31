#!/usr/bin/env bash
echo "Google engine 1"
echo "processing:"
ssh -T -i ~/.ssh/server1 weiziyang@35.225.41.22 "ps -ef|grep correction|awk 'NR==1'"|awk -F "correction.py" '{print $2}'
echo "completed:"
ssh -T -i ~/.ssh/server1 weiziyang@35.225.41.22 "ls /home/g5835udock0/n4_bias/"
echo "  "

echo "Google engine 2"
echo "processing:"
ssh -T -i ~/.ssh/server1 weiziyang@34.68.198.238 "ps -ef|grep correction|awk 'NR==1'"|awk -F "correction.py" '{print $2}'
echo "completed:"
ssh -T -i ~/.ssh/server1 weiziyang@34.68.198.238 "ls /home/g5835udock0/n4_bias/"
echo "  "

echo "Google engine 3"
echo "processing:"
ssh -T -i ~/.ssh/server1 weiziyang@35.232.10.92 "ps -ef|grep correction|awk 'NR==1'"|awk -F "correction.py" '{print $2}'
echo "completed:"
ssh -T -i ~/.ssh/server1 weiziyang@35.232.10.92 "ls /home/g5835udock0/n4_bias/"
echo "  "

echo "Google engine 4"
echo "preprocessing:"
ssh -T -i ~/.ssh/server1 weiziyang@104.154.151.164 "ps -ef|grep correction|awk 'NR==1'"|awk -F "correction.py" '{print $2}'
echo "completed:"
ssh -T -i ~/.ssh/server1 weiziyang@104.154.151.164 "ls /home/g5835udock0/n4_bias/"
echo "  "


echo "Ali Clound"
echo "processing:"
ssh -T root@106.15.227.178 -p 22 -i ~/.ssh/server1  "ps -ef|grep correction|awk 'NR==1'"|awk -F "correction.py" '{print $2}'
echo "completed:"
ssh -T root@106.15.227.178 -p 22 -i ~/.ssh/server1 "ls /root/final_project/n4_bias/"
echo "  "

echo "Vultr"
echo "processing:"
ssh -T root@167.179.111.96 -p 22 -i ~/.ssh/server1  "ps -ef|grep correction|awk 'NR==1'"|awk -F "correction.py" '{print $2}'
echo "completed:"
ssh -T root@167.179.111.96 -p 22 -i ~/.ssh/server1 "ls /root/final_project/n4_bias/"
echo "  "

ssh  -i ~/.ssh/server1 weiziyang@35.226.35.10

104.154.151.164
