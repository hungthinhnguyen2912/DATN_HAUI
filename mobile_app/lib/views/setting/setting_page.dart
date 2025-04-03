import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../App_Color.dart';
import '../../P.dart';
import 'items_setting_page/edit_profile_page.dart';

class SettingPage extends StatefulWidget {
  const SettingPage({super.key});

  @override
  State<SettingPage> createState() => _SettingPageState();
}

class _SettingPageState extends State<SettingPage> {
  bool isLoading = false;

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
        centerTitle: true,
      ),
      body:
          isLoading
              ? const Center(child: CircularProgressIndicator())
              : SingleChildScrollView(
                child: Column(
                  children: [
                    const SizedBox(height: 30),
                    _buildProfileSection(),
                    const SizedBox(height: 20),
                    _buildDivider(),
                    _buildAccountSettings(),
                    const SizedBox(height: 20),
                    _buildDivider(),
                    _buildAppSettings(),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
    );
  }

  /// 🎨 Thiết kế giao diện Profile Section
  Widget _buildProfileSection() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: [
          Stack(
            children: [
              CircleAvatar(
                radius: 45,
                backgroundColor: Colors.white,
                child: Obx(() {
                  if (P.avatar.avatarUrl.value.isEmpty) {
                    return const Icon(
                      Icons.account_circle,
                      size: 80,
                      color: Colors.black,
                    );
                  } else {
                    return ClipOval(
                      child: Image.network(
                        P.avatar.avatarUrl.value,
                        width: 90,
                        height: 90,
                        fit: BoxFit.cover,
                      ),
                    );
                  }
                }),
              ),
              Positioned(
                bottom: 0,
                right: 0,
                child: GestureDetector(
                  onTap: _showAvatarOptions,
                  child: Container(
                    padding: const EdgeInsets.all(6),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: AppColors.green,
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black26,
                          blurRadius: 4,
                          spreadRadius: 2,
                        ),
                      ],
                    ),
                    child: const Icon(
                      Icons.camera_alt,
                      color: Colors.white,
                      size: 20,
                    ),
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

  void _showAvatarOptions() {
    showModalBottomSheet(
      context: context,
      builder: (context) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildBottomSheetItem(Icons.camera_alt, "Camera", () async {
                await P.avatar.cameraAvatar();
                await P.avatar.postCloudinary();
              }),
              _buildBottomSheetItem(Icons.image, "Gallery", () async {
                await P.avatar.galleryAvatar();
                await P.avatar.postCloudinary();
              }),
            ],
          ),
        );
      },
    );
  }

  Widget _buildBottomSheetItem(IconData icon, String text, VoidCallback onTap) {
    return ListTile(leading: Icon(icon), title: Text(text), onTap: onTap);
  }

  Widget _buildDivider() {
    return Divider(
      thickness: 1.2,
      color: AppColors.green.withOpacity(0.7),
      indent: 30,
      endIndent: 30,
    );
  }

  Widget _buildSettingItem(IconData icon, String title, VoidCallback onTap) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 2,
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
          child: Row(
            children: [
              Icon(icon, size: 26, color: AppColors.green),
              const SizedBox(width: 16),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
              const Icon(Icons.arrow_forward_ios, size: 18, color: Colors.grey),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAccountSettings() {
    return Column(
      children: [
        _buildSettingItem(
          Icons.person,
          "Edit Profile",
          () => Get.to(() => EditProfilePage()),
        ),
        _buildSettingItem(Icons.key, "Change Password", () {}),
      ],
    );
  }

  Widget _buildAppSettings() {
    return Column(
      children: [
        _buildSettingItem(Icons.lock_clock, "Clear History", _clearHistory),
        _buildSettingItem(Icons.delete_forever, "Delete Account", () {}),
        _buildSettingItem(Icons.logout, "Log Out", () => P.auth.logOut()),
      ],
    );
  }

  Future<void> _clearHistory() async {
    Get.dialog(
      AlertDialog(
        title: const Text("Clear History"),
        content: const Text("Are you sure you want to clear your history?"),
        actions: [
          TextButton(
            onPressed: () async {
              setState(() => isLoading = true);
              Get.back();
              await P.history.clearHistory();
              setState(() => isLoading = false);
              Get.snackbar(
                "Success",
                "History cleared!",
                snackPosition: SnackPosition.BOTTOM,
              );
            },
            child: const Text("Yes"),
          ),
          TextButton(onPressed: () => Get.back(), child: const Text("No")),
        ],
      ),
    );

  }
}
