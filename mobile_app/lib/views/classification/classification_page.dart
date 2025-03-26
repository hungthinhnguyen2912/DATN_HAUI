import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:mobile_app/App_Color.dart';
import 'package:get/get.dart';
import 'package:mobile_app/components/button.dart';
import '../../P.dart';
import '../../models/History.dart';

class ClassificationPage extends StatelessWidget {
  const ClassificationPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Classification",
          style: TextStyle(
            color: AppColors.text_color,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: false,
        backgroundColor: AppColors.green,
      ),
      body: Center(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Expanded(
              flex: 2,
              child:
                  P.image.image.value != null
                      ? Obx(() {
                        return Image.file(
                          P.image.image.value!,
                          width: 300,
                          height: 300,
                        );
                      })
                      : Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.center,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Container(
                            width: 300,
                            height: 300,
                            decoration: BoxDecoration(
                              color: Colors.grey,
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(color: Colors.grey, width: 2),
                            ),
                          ),
                          Obx(() {
                            return Text(
                              "Result: ${P.classification.result.value}",
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: AppColors.green,
                                fontSize: 30,
                              ),
                            );
                          }),
                        ],
                      ),
            ),
            Expanded(
              flex: 1,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ButtonClassification(
                    icon: Icon(Icons.image_search, color: AppColors.white),
                    content: "Classify",
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
                      await P.image.uploadToCloudinary();
                      await P.classification.classifyImage(
                        P.image.image.value!,
                      );
                      History his = History(
                        createdAt: Timestamp.now(),
                        kind: P.classification.result.value,
                        imageUrl: P.image.imageUrl.value,
                        uidUser:
                            P.auth.currentUser.value?.name ?? "Unknown User",
                      );
                      await P.classification.postHistory(his);
                    },
                  ),
                  ButtonClassification(
                    icon: Icon(Icons.upload_file, color: AppColors.white),
                    content: "Upload picture from gallery",
                    onTap: () {
                      P.image.galleryImage();
                    },
                  ),

                  ButtonClassification(
                    icon: Icon(Icons.camera, color: AppColors.white),
                    content: "Take a picture",
                    onTap: () {
                      P.image.cameraImage();
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
