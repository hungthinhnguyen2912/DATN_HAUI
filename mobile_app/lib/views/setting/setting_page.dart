import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../../App_Color.dart';
import '../../P.dart';

class SettingPage extends StatelessWidget {
  const SettingPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppColors.green,
        title: Text(
          "Setting",
          style: TextStyle(
            color: AppColors.text_color,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Column(
        children: [
          SizedBox(height: 30,),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircleAvatar(
                radius: 30,
                child: Icon(Icons.account_circle_outlined,size: 40,),
              ),
              SizedBox(width: 20,),
              Column(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Obx(
                    () => Text(
                      P.auth.currentUser.value?.name ?? "Unknown",
                      style: TextStyle(fontSize: 25),
                    ),
                  ),
                  Obx(
                    () => Text(
                      P.auth.currentUser.value?.email ?? "Unknown email",
                    ),
                  ),
                ],
              ),
            ],
          ),
          SizedBox(height: 20,),
          Divider(
            thickness: 1,
            color: AppColors.green,
            endIndent: 27.98,
            indent: 27.98,
          ),
          SizedBox(height: 20,),
          _buildItem(Icons.person, "Edit Profile" ,() {}),
          _buildItem(Icons.key, "Change password",() {}),
          SizedBox(height: 30 ,),
          Divider(
            thickness: 1,
            color: AppColors.green,
            endIndent: 27.98,
            indent: 27.98,
          ),
          _buildItem(Icons.notifications, "Notification", () {}),
          _buildItem(Icons.language, "Language", () {}),
          _buildItem(Icons.logout, "Logout", () {
            P.auth.logOut();
          }),
        ],
      ),
    );
  }
  Widget _buildItem(IconData icon, String title, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10),
          color: Colors.white,
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.4),
              blurRadius: 5,
              spreadRadius: 2,
              offset: Offset(0, 3),
            ),
          ],
        ),
        child: Row(
          children: [
            Icon(icon, size: 28, color: Colors.green),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                title,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  color: Colors.black87,
                ),
              ),
            ),
            Icon(Icons.arrow_forward_ios, size: 20, color: Colors.grey),
          ],
        ),
      ),
    );
  }

}
