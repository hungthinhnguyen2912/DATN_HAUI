import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:mobile_app/App_Color.dart';
import 'package:mobile_app/components/button.dart';
import 'package:mobile_app/controllers/img_controller.dart';
import '../../P.dart';
import '../../models/History.dart';

class ClassificationPage extends StatefulWidget {
  const ClassificationPage({super.key});

  @override
  State<ClassificationPage> createState() => _ClassificationPageState();
}

class _ClassificationPageState extends State<ClassificationPage> {
  bool _isLoading = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Text(
          "Classification",
          style: TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: false,
        backgroundColor: AppColors.green,
        elevation: 4,
        shadowColor: Colors.black54,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Obx(() {
                return Container(
                  width: 250,
                  height: 250,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(20),
                    color: Colors.grey[300],
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black26,
                        blurRadius: 10,
                        spreadRadius: 2,
                        offset: Offset(0, 4),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(20),
                    child:
                        P.image.image.value != null
                            ? Image.file(
                              P.image.image.value!,
                              width: 250,
                              height: 250,
                              fit: BoxFit.cover,
                            )
                            : Center(
                              child: Icon(
                                Icons.image,
                                size: 100,
                                color: Colors.grey[600],
                              ),
                            ),
                  ),
                );
              }),
              const SizedBox(height: 20),
              Obx(() {
                return Text(
                  "Result: ${P.classification.result.value}",
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: AppColors.green,
                    fontSize: 24,
                  ),
                );
              }),
              const SizedBox(height: 30),
              if (_isLoading) CircularProgressIndicator(color: AppColors.green),
              if (!_isLoading)
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ButtonClassification(
                        icon: Icons.image_search,
                        label: "Classify",
                        onTap: () async {
                          if (P.image.image.value == null) {
                            Get.snackbar(
                              "Error",
                              "Please select an image first!",
                              backgroundColor: Colors.red,
                              colorText: Colors.white,
                            );
                            return;
                          }
                          setState(() {
                            _isLoading = true;
                          });
                          await P.image.postImgCloudinary();
                          await P.classification.classifyImage(
                            P.image.image.value!,
                          );
                          History his = History(
                            createdAt: Timestamp.fromDate(DateTime.now()),
                            kind: P.classification.result.value,
                            imageUrl: P.image.imageUrl.value,
                            uidUser:
                                P.auth.currentUser.value?.uid ?? "Unknown User",
                            publicId: P.image.publicId.value,
                          );
                          await P.classification.postHistory(his);
                          setState(() {
                            _isLoading = false;
                          });
                        },
                      ),
                      ButtonClassification(
                        icon: Icons.upload_file,
                        label: "Upload from Gallery",
                        onTap: () {
                          P.image.galleryImage();
                        },
                      ),
                      ButtonClassification(
                        icon: Icons.camera_alt,
                        label: "Take a Picture",
                        onTap: () {
                          P.image.cameraImage();
                        },
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}
