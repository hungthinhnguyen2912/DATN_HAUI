import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../../../App_Color.dart';
import '../../../P.dart';
import '../../../components/button.dart';
import '../../../components/input_text_field.dart';
import '../auth_screen.dart';


class LogInWithPhonePage extends StatelessWidget {
  LogInWithPhonePage({super.key});

  final TextEditingController _phoneController = TextEditingController();

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
                  "Sign in with phone",
                  style: TextStyle(
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                    fontSize: 20,
                  ),
                ),
                SizedBox(height: 18),
                Text(
                  "Enter your phone number to verify your account",
                  style: TextStyle(color: AppColors.gray_login_text),
                ),
              ],
            ),
          ),
          InputTextField(
            controller: _phoneController,
            textWarning: "Enter your phone number",
            hintText: "Enter your phone number",
            obs: false, readOnly: false,
          ),
          Padding(
            padding: const EdgeInsets.only(left: 20),
            child: ButtonAuth(
              content: "Verify phone number",
              onTap: () {
              },
            ),
          ),
        ],
      ),
    );
  }
}
