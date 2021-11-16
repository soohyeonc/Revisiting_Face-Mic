//Created by Dilmer Valecillos, amended by Alex Coulombe @ibrews to signal presses and releases and log controller

using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR;
using DilmerGamesLogger = DilmerGames.Core.Logger;
using System.IO;
using System.Text;
using System;



public class InputManager : MonoBehaviour
{
    [SerializeField]
    private XRNode xRNode_head = XRNode.Head;
    //private XRNode xRNode_lefthand = XRNode.LeftHand;
    private XRNode xRNode_righthand = XRNode.RightHand;


    private List<InputDevice> devices = new List<InputDevice>();
    private InputDevice device;

    private List<InputDevice> right_devices = new List<InputDevice>();
    private InputDevice right_device;
    
    private bool devicePositionChosen;
    private Vector3 devicePositionValue = Vector3.zero;
    private Vector3 prevdevicePositionValue;

    private bool deviceVelocityChosen;
    private Vector3 deviceVelocityValue = Vector3.zero;
    private Vector3 prevdeviceVelocityValue;

    private bool deviceAccelChosen;
    private Vector3 deviceAccelValue = Vector3.zero;
    private Vector3 prevdeviceAccelValue;

    private bool deviceRotateChosen;
    private Quaternion deviceRotateValue;
    private Quaternion prevdeviceRotateValue;

    private List<string[]> data = new List<string[]>();


    bool first_time = false;
    float ft;

    void Start()
    {
        string path = Application.persistentDataPath + "/sensordata.csv";
        string[] temp_data = new string[15];
        temp_data[0] = "Time_s_";
        temp_data[1] = "Acc_x";
        temp_data[2] = "Acc_y";
        temp_data[3] = "Acc_z";
        temp_data[4] = "Gyro_x";
        temp_data[5] = "Gyro_y";
        temp_data[6] = "Gyro_z";
        temp_data[7] = "Velo_x";
        temp_data[8] = "Velo_y";
        temp_data[9] = "Velo_z";
        temp_data[10] = "Pos_x";
        temp_data[11] = "Pos_y";
        temp_data[12] = "Pos_z";
        temp_data[13] = "sum_xyz";
        temp_data[14] = "magnitude";
        data.Add(temp_data);
    }
    void GetDevice()
    {
        InputDevices.GetDevicesAtXRNode(xRNode_head, devices);
        device = devices.FirstOrDefault();

        InputDevices.GetDevicesAtXRNode(xRNode_righthand, right_devices);
        right_device = right_devices.FirstOrDefault();
    }

    /*
    void OnEnable()
    {
        if (!device.isValid)
        {
            GetDevice();
        }
    }
    */

    void Update()
    {
        if (!device.isValid)
        {
            GetDevice();
        }

        InputFeatureUsage<Vector3> deviceVelocityUsage = CommonUsages.deviceVelocity;
        InputFeatureUsage<Vector3> devicePositionsUsage = CommonUsages.devicePosition;
        InputFeatureUsage<Vector3> deviceAccelUsage = CommonUsages.deviceAcceleration;
        InputFeatureUsage<Quaternion> deviceRotateUsage = CommonUsages.deviceRotation;

        bool triggerButtonValue = false;
        if (right_device.TryGetFeatureValue(CommonUsages.triggerButton, out triggerButtonValue) && triggerButtonValue)
        {
            
            DilmerGamesLogger.Instance.LogInfo($"TriggerButton activated {triggerButtonValue} on {xRNode_righthand}");


            //potision value     
            if (devicePositionValue != prevdevicePositionValue)
            {
                devicePositionChosen = false;
            }

            if (device.TryGetFeatureValue(devicePositionsUsage, out devicePositionValue) && devicePositionValue != Vector3.zero && !devicePositionChosen)
            {
                DilmerGamesLogger.Instance.LogInfo($"Position value on {xRNode_head} = {devicePositionValue.ToString("F3")}");
                prevdevicePositionValue = devicePositionValue;
                devicePositionChosen = true;
            }
            else if (devicePositionValue == Vector3.zero && devicePositionChosen)
            {
                DilmerGamesLogger.Instance.LogInfo($"Position value on {xRNode_head} = {devicePositionValue.ToString("F3")}");
                prevdevicePositionValue = devicePositionValue;
                devicePositionChosen = false;
            }


            //velocity value
            if (deviceVelocityValue != prevdeviceVelocityValue)
            {
                deviceVelocityChosen = false;
            }

            if (device.TryGetFeatureValue(deviceVelocityUsage, out deviceVelocityValue) && deviceVelocityValue != Vector3.zero && !deviceVelocityChosen)
            {
                DilmerGamesLogger.Instance.LogInfo($"Position value on {xRNode_head} = {devicePositionValue.ToString("F3")}");
                prevdeviceVelocityValue = deviceVelocityValue;
                deviceVelocityChosen = true;
            }
            else if (deviceVelocityValue == Vector3.zero && deviceVelocityChosen)
            {
                DilmerGamesLogger.Instance.LogInfo($"Position value on {xRNode_head} = {devicePositionValue.ToString("F3")}");
                prevdeviceVelocityValue = deviceVelocityValue;
                deviceVelocityChosen = false;
            }

            //acceleration value
            if (deviceAccelValue != prevdeviceAccelValue)
            {
                deviceAccelChosen = false;
            }

            if (device.TryGetFeatureValue(deviceAccelUsage, out deviceAccelValue) && deviceAccelValue != Vector3.zero && !deviceAccelChosen)
            {
                DilmerGamesLogger.Instance.LogInfo($"Acceleration value on {xRNode_head} = {deviceAccelValue.ToString("F3")}");
                prevdeviceAccelValue = deviceAccelValue;
                deviceAccelChosen = true;
            }
            else if (deviceAccelValue == Vector3.zero && deviceAccelChosen)
            {
                DilmerGamesLogger.Instance.LogInfo($"Acceleration value on {xRNode_head} = {deviceAccelValue.ToString("F3")}");
                prevdeviceAccelValue = deviceAccelValue;
                deviceAccelChosen = false;
            }

            //Rotation value
            if (deviceRotateValue != prevdeviceRotateValue)
            {
                deviceRotateChosen = false;
            }


            if (device.TryGetFeatureValue(deviceRotateUsage, out deviceRotateValue) && !deviceRotateChosen)
            {
                DilmerGamesLogger.Instance.LogInfo($"Rotation value on {xRNode_head} = {deviceRotateValue.ToString("F3")}");
                prevdeviceRotateValue = deviceRotateValue;
                deviceRotateChosen = true;
            }
            else if (deviceRotateChosen)
            {
                DilmerGamesLogger.Instance.LogInfo($"Rotation value on {xRNode_head} = {deviceRotateValue.ToString("F3")}");
                prevdeviceRotateValue = deviceRotateValue;
                deviceRotateChosen = false;
            }


            //time setting

            float timeChangeInMillis;
            if (first_time == false)
            {
                ft = Time.realtimeSinceStartup;
                timeChangeInMillis = Time.realtimeSinceStartup-ft;
                first_time = true;
            }
            else
            {
                timeChangeInMillis = Time.realtimeSinceStartup - ft;
            }
            
            
            //how to increase sampling rate.
            
           
            //write CSV file 


            string[] temp_data = new string[15];
            //time
            temp_data[0] = timeChangeInMillis.ToString();
            //acc
            temp_data[1] = deviceAccelValue.x.ToString();
            temp_data[2] = deviceAccelValue.y.ToString();
            temp_data[3] = deviceAccelValue.z.ToString();
            //gyro
            temp_data[4] = deviceRotateValue.x.ToString();
            temp_data[5] = deviceRotateValue.y.ToString();
            temp_data[6] = deviceRotateValue.z.ToString();
            //velocity
            temp_data[7] = deviceVelocityValue.x.ToString();
            temp_data[8] = deviceVelocityValue.y.ToString();
            temp_data[9] = deviceVelocityValue.z.ToString();
            //position
            temp_data[10] = devicePositionValue.x.ToString();
            temp_data[11] = devicePositionValue.y.ToString();
            temp_data[12] = devicePositionValue.z.ToString();
            //summed xyz
            temp_data[13] = (deviceAccelValue.x + deviceAccelValue.y + deviceAccelValue.z).ToString();
            //magnitude
            temp_data[14] = ( Math.Sqrt(Math.Pow(deviceAccelValue.x, 2) + Math.Pow(deviceAccelValue.y ,2) + Math.Pow(deviceAccelValue.z,2)) ).ToString();
            data.Add(temp_data);


        }
        else if (!triggerButtonValue)
        {
            if (first_time)
            {
                //string path = Application.persistentDataPath + "/sensordata.csv";
                string[][] output = new string[data.Count][];
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = data[i];
                }
                int length = output.GetLength(0);
                string delimiter = ",";
                StringBuilder sb = new StringBuilder();
                for (int index = 0; index < length; index++)
                    sb.AppendLine(string.Join(delimiter, output[index]));
                string filePath = path;

                StreamWriter outStream = System.IO.File.CreateText(filePath);
                outStream.WriteLine(sb);
                outStream.Close();
            }
            first_time = false;


        }

    }
}
