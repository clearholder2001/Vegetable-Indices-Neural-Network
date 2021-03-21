# Vegetable-Indices-Neural-Network
Artificial Neural Network-based Vegetable Indices  
*National Cheng Kung University, Taiwan*  
*Ministry of Science and Technology, Taiwan*

# Information
This repo is the ongoing part of the project "UAV-based Monitoring and Recognition System for Pest Insects in Organic Vegetables". Please switch different branch to check development progress.

# Project Abstract
The development and technology of unmanned aerial vehicle (UAV) are maturing. The UAV advantages of lightweight and high mobility can facilitate efficient farm monitoring, which can efficiently reduce the manpower in crop monitoring. The precise farming information can be obtained from the analysis of crop images, which can facilitate the crop pest management, reducing the requirement of manpower and increasing productivity. The goal of this project is to establish a UAV-based monitoring and recognition system for pest insects in organic vegetables. Considering the flexibility on farmland types, a lightweight UAV equipped with one-inch CMOS (Complementary Metal Oxide Semiconductor) sensor. A UAV with camera sensor is used to acquire images of organic vegetables and sticky papers set up in the study area. In case of small-size farmland, a fixed and lightweight track equipped with a camera sensor can be installed. To monitor the growth of organic vegetables from the acquired images and to distinguish organic vegetables from sticky papers, a deep-learning-based vegetation index is proposed in which the network structure is designed based on residual network. The vegetation index is used to monitor the vegetation growth in the net farmland. In addition, a light and efficient-learning version of YOLO (You only Look Once), called YOLO Nano, is utilized to detect sticky papers and pest insects. The detection accuracy for the sticky papers and pest insects are 98.5% and 64.8%, respectively. The system is developed based on OpenDroneMap, which is an open source photogrammetry toolkit to process aerial imagery into maps. With the orthoimages in the system, the users can intuitively access the geospatial information.