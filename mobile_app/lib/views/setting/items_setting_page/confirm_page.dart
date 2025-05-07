import 'package:flutter/material.dart';
import 'package:mobile_app/App_Color.dart';
import 'package:mobile_app/components/button.dart';
import 'package:mobile_app/components/input_text_field.dart';
import 'package:get/get.dart';
import '../../../P.dart';

class ConfirmPage extends StatelessWidget {
  ConfirmPage({super.key});

  final TextEditingController _passController = TextEditingController();
  final TextEditingController _confirmPassController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppColors.green,
        title: Text(
          "Change Password",
          style: TextStyle(color: AppColors.white, fontSize: 20),
        ),
        centerTitle: true,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(height: 20),
          InputTextField(
            controller: _passController,
            textWarning: "Enter your new password",
            hintText: "New password",
            obs: true,
            readOnly: false,
          ),
          SizedBox(height: 20),
          InputTextField(
            controller: _confirmPassController,
            textWarning: "Confirm your new password",
            hintText: "Confirm new password",
            obs: true,
            readOnly: false,
          ),
          ButtonAuth(
            content: "Change",
            onTap: () {
              if (_passController.text == _confirmPassController.text) {
                P.auth.updatePassword(_passController.text);
              } else {
                Get.snackbar("Error", "Mật khẩu không khớp.");
              }
            },
          ),
        ],
      ),
    );
  }
}
