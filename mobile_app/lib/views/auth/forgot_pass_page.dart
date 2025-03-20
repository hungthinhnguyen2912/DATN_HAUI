import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../../App_Color.dart';
import '../../components/button.dart';
import '../../components/input_text_field.dart';
import 'auth_screen.dart';

class ForgotPassPage extends StatelessWidget {
  ForgotPassPage({super.key});

  final TextEditingController _forgotPassController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(left: 25, top: 83, right: 25),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: Color(0xFFECECEC),
                    borderRadius: BorderRadius.circular(42),
                  ),
                  child: Center(
                    child: IconButton(
                      onPressed: () {
                        Get.off(() => AuthScreen());
                      },
                      icon: Icon(Icons.arrow_back_ios),
                    ),
                  ),
                ),
                SizedBox(height: 26),
                Text(
                  "Forgot password",
                  style: TextStyle(
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                    fontSize: 20,
                  ),
                ),
                SizedBox(height: 18),
                Text(
                  "Please enter your email to reset your password",
                  style: TextStyle(color: AppColors.gray_login_text),
                ),
              ],
            ),
          ),
          InputTextField(
            controller: _forgotPassController,
            textWarning: "Enter your email",
            hintText: "Enter your email",
            obs: false,
          ),
          Padding(
            padding: const EdgeInsets.only(left: 20),
            child: ButtonAuth(
              content: "Reset password",
              onTap: () {},
            ),
          ),
        ],
      ),
    );
  }
}
