#!/bin/bash


echo "############## Attack 1 s iterations ############"
for i in {1..5}
do
  echo "############## $i s iterations ############"
  python3 Multi_CNN_8users_attack1.py
done

echo "############## Attack 2 s iterations ############"
for i in {1..5}
do
  echo "############## $i s iterations ############"
  python3 Multi_CNN_8users_attack2.py
done

echo "############## Attack 3 s iterations ############"
for i in {1..5}
do
  echo "############## $i s iterations ############"
  python3 Multi_CNN_8users_attack3.py
done

echo "############## Attack 4 s iterations ############"
for i in {1..5}
do
  echo "############## $i s iterations ############"
  python3 Multi_CNN_8users_attack4.py
done

echo "############## Attack 5 s iterations ############"
for i in {1..5}
do
  echo "############## $i s iterations ############"
  python3 Multi_CNN_8users_attack5.py
done

echo "############## Attack 6 s iterations ############"
for i in {1..5}
do
  echo "############## $i s iterations ############"
  python3 Multi_CNN_8users_attack6.py
done

echo "############## Attack 8 s iterations ############"
for i in {1..5}
do
  echo "############## $i s iterations ############"
  python3 Multi_CNN_8users_attack7.py
done
