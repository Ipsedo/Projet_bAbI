#!/usr/bin/env bash

if ! [ -d "./res" ]; then
    mkdir res
fi

cd res
if ! [ -d "./tasks_1-20_v1-2" ]; then
    wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
    tar -xvzf tasks_1-20_v1-2.tar.gz
    rm -rf tasks_1-20_v1-2.tar.gz
fi
cd ..