import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:mobile_app/views/setting/items_setting_page/edit_profile_page.dart';

import '../../App_Color.dart';
import '../../P.dart';

class SettingPage extends StatefulWidget {
  const SettingPage({super.key});

  @override
  State<SettingPage> createState() => _SettingPageState();
}

class _SettingPageState extends State<SettingPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        backgroundColor: AppColors.green,
        title: Text(
          "Settings",
          style: TextStyle(
            color: AppColors.text_color,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: ListView(
        children: [
          const SizedBox(height: 30),
          _buildProfileSection(),
          const SizedBox(height: 20),
          _buildDivider(),
          const SizedBox(height: 20),
          _buildAccountSettings(),
          const SizedBox(height: 30),
          _buildDivider(),
          _buildAppSettings(),
        ],
      ),
    );
  }

  Widget _buildProfileSection() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          Stack(
            children: [
              Container(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 8,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: CircleAvatar(
                  radius: 45,
                  backgroundColor: Colors.white,
                  child: Obx(() {
                    if (P.avatar.avatarUrl.value == "") {
                      return const Icon(Icons.account_circle, size: 80, color: Colors.black);
                    } else {
                      return CircleAvatar(
                        radius: 24,
                        backgroundImage: NetworkImage(P.avatar.avatarUrl.value),
                      );
                    }
                  }),
                ),
              ),
              Positioned(
                bottom: 0,
                right: 0,
                width: 30,
                height: 30,
                child: Container(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppColors.green,
                    boxShadow: const [
                      BoxShadow(
                        color: Colors.black26,
                        blurRadius: 4,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: IconButton(
                    icon: const Icon(
                      Icons.camera_alt,
                      color: Colors.white,
                      size: 19,
                    ),
                    padding: const EdgeInsets.all(4),
                    constraints: const BoxConstraints(),
                    onPressed: () {
                      print("Thêm avatar");
                      showModalBottomSheet(context: context, builder: (context) {
                        return SafeArea(
                          child: Container(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Card(
                                  elevation: 2,
                                  child: ListTile(
                                    leading: const Icon(Icons.camera_alt),
                                    title: const Text("Camera"),
                                    onTap: () async {
                                      await P.avatar.cameraAvatar();
                                      await P.avatar.postAvatarToCloudinary();
                                    }
                                  ),
                                ),
                                Card(
                                  elevation: 2,
                                  child: ListTile(
                                    leading: const Icon(Icons.image),
                                    title: Text("Gallery"),
                                    onTap: () async {
                                      await P.avatar.galleryAvatar();
                                      await P.avatar.postAvatarToCloudinary();
                                    },
                                  ),
                                )
                              ],
                            ),
                          ),
                        );
                      });
                    },
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(width: 20),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Obx(
                () => Text(
                  P.auth.currentUser.value?.name ?? "Unknown",
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 5),
              Obx(
                () => Text(
                  P.auth.currentUser.value?.email ?? "Unknown email",
                  style: TextStyle(color: Colors.grey[700]),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildDivider() {
    return Divider(
      thickness: 1.5,
      color: AppColors.green.withOpacity(0.7),
      endIndent: 30,
      indent: 30,
    );
  }

  Widget _buildSettingItem(IconData icon, String title, VoidCallback onTap) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 3,
      child: ListTile(
        onTap: onTap,
        leading: Icon(icon, size: 28, color: AppColors.green),
        title: Text(
          title,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
        ),
        trailing: const Icon(
          Icons.arrow_forward_ios,
          size: 20,
          color: Colors.grey,
        ),
      ),
    );
  }

  Widget _buildAccountSettings() {
    return Column(
      children: [
        _buildSettingItem(Icons.person, "Edit Profile", () {
          Get.to(EditProfilePage());
        }),
        _buildSettingItem(Icons.key, "Change Password", () {}),
      ],
    );
  }

  Widget _buildAppSettings() {
    return Column(
      children: [
        _buildSettingItem(Icons.lock_clock, "Clear History", () {}),
        _buildSettingItem(Icons.delete_forever, "Delete Account", () {}),
        _buildSettingItem(Icons.logout, "Log Out", () {
          P.auth.logOut();
        }),
      ],
    );
  }
}
