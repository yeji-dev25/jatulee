package com.p_project.writing;


import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/writing")
@RequiredArgsConstructor
public class WritingSessionController {

    private final WritingSessionService writingService;


    // 일기 완료 처리
    @PostMapping("/{id}/complete")
    public ResponseEntity<?> complete(@PathVariable Long id) {
        return ResponseEntity.ok(writingService.complete(id));
    }
}