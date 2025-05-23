import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:mobile_app/App_Color.dart';
import 'package:mobile_app/components/input_text_field.dart';
import '../../../P.dart';

class EditProfilePage extends StatelessWidget {
  EditProfilePage({super.key});

  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _userNameController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Edit Profile',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 20,
            color: AppColors.white,
          ),
        ),
        centerTitle: true,
        elevation: 0,
        backgroundColor: AppColors.green,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 30),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 20),
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              child: Padding(
                padding: const EdgeInsets.all(10.0),
                child: InputTextField(
                  controller: _emailController,
                  textWarning: "",
                  hintText:
                      P.auth.currentUser.value!.email,
                  obs: false,
                  readOnly: false,
                  // prefixIcon: const Icon(Icons.email, color: Colors.blueAccent),
                ),
              ),
            ),
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              child: Padding(
                padding: const EdgeInsets.all(10.0),
                child: InputTextField(
                  controller: _userNameController,
                  textWarning: "",
                  hintText:
                      P.auth.currentUser.value!.name ?? 'Email not available',
                  obs: false,
                  readOnly: false,
                  // prefixIcon: const Icon(Icons.email, color: Colors.blueAccent),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Center(
              child: ElevatedButton(
                onPressed: () {
                  Get.snackbar('Success', 'Profile updated successfully!');
                  P.auth.editProfile(
                    _emailController.text,
                    _userNameController.text,
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 40,
                    vertical: 15,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: const Text(
                  'Save Changes',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ],
        ),
      ),
      backgroundColor: Colors.grey[100],
    );
  }
}
